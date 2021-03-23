// <auto-generated>
//     Generated by the protocol buffer compiler.  DO NOT EDIT!
//     source: tensorflow/core/framework/device_attributes.proto
// </auto-generated>
#pragma warning disable 1591, 0612, 3021
#region Designer generated code

using pb = global::Google.Protobuf;
using pbc = global::Google.Protobuf.Collections;
using pbr = global::Google.Protobuf.Reflection;
using scg = global::System.Collections.Generic;
namespace Tensorflow {

  /// <summary>Holder for reflection information generated from tensorflow/core/framework/device_attributes.proto</summary>
  public static partial class DeviceAttributesReflection {

    #region Descriptor
    /// <summary>File descriptor for tensorflow/core/framework/device_attributes.proto</summary>
    public static pbr::FileDescriptor Descriptor {
      get { return descriptor; }
    }
    private static pbr::FileDescriptor descriptor;

    static DeviceAttributesReflection() {
      byte[] descriptorData = global::System.Convert.FromBase64String(
          string.Concat(
            "CjF0ZW5zb3JmbG93L2NvcmUvZnJhbWV3b3JrL2RldmljZV9hdHRyaWJ1dGVz",
            "LnByb3RvEgp0ZW5zb3JmbG93IkUKEEludGVyY29ubmVjdExpbmsSEQoJZGV2",
            "aWNlX2lkGAEgASgFEgwKBHR5cGUYAiABKAkSEAoIc3RyZW5ndGgYAyABKAUi",
            "OAoKTG9jYWxMaW5rcxIqCgRsaW5rGAEgAygLMhwudGVuc29yZmxvdy5JbnRl",
            "cmNvbm5lY3RMaW5rIloKDkRldmljZUxvY2FsaXR5Eg4KBmJ1c19pZBgBIAEo",
            "BRIRCgludW1hX25vZGUYAiABKAUSJQoFbGlua3MYAyABKAsyFi50ZW5zb3Jm",
            "bG93LkxvY2FsTGlua3MirAEKEERldmljZUF0dHJpYnV0ZXMSDAoEbmFtZRgB",
            "IAEoCRITCgtkZXZpY2VfdHlwZRgCIAEoCRIUCgxtZW1vcnlfbGltaXQYBCAB",
            "KAMSLAoIbG9jYWxpdHkYBSABKAsyGi50ZW5zb3JmbG93LkRldmljZUxvY2Fs",
            "aXR5EhMKC2luY2FybmF0aW9uGAYgASgGEhwKFHBoeXNpY2FsX2RldmljZV9k",
            "ZXNjGAcgASgJQpEBChhvcmcudGVuc29yZmxvdy5mcmFtZXdvcmtCFkRldmlj",
            "ZUF0dHJpYnV0ZXNQcm90b3NQAVpYZ2l0aHViLmNvbS90ZW5zb3JmbG93L3Rl",
            "bnNvcmZsb3cvdGVuc29yZmxvdy9nby9jb3JlL2ZyYW1ld29yay9kZXZpY2Vf",
            "YXR0cmlidXRlc19nb19wcm90b/gBAWIGcHJvdG8z"));
      descriptor = pbr::FileDescriptor.FromGeneratedCode(descriptorData,
          new pbr::FileDescriptor[] { },
          new pbr::GeneratedClrTypeInfo(null, null, new pbr::GeneratedClrTypeInfo[] {
            new pbr::GeneratedClrTypeInfo(typeof(global::Tensorflow.InterconnectLink), global::Tensorflow.InterconnectLink.Parser, new[]{ "DeviceId", "Type", "Strength" }, null, null, null, null),
            new pbr::GeneratedClrTypeInfo(typeof(global::Tensorflow.LocalLinks), global::Tensorflow.LocalLinks.Parser, new[]{ "Link" }, null, null, null, null),
            new pbr::GeneratedClrTypeInfo(typeof(global::Tensorflow.DeviceLocality), global::Tensorflow.DeviceLocality.Parser, new[]{ "BusId", "NumaNode", "Links" }, null, null, null, null),
            new pbr::GeneratedClrTypeInfo(typeof(global::Tensorflow.DeviceAttributes), global::Tensorflow.DeviceAttributes.Parser, new[]{ "Name", "DeviceType", "MemoryLimit", "Locality", "Incarnation", "PhysicalDeviceDesc" }, null, null, null, null)
          }));
    }
    #endregion

  }
  #region Messages
  public sealed partial class InterconnectLink : pb::IMessage<InterconnectLink> {
    private static readonly pb::MessageParser<InterconnectLink> _parser = new pb::MessageParser<InterconnectLink>(() => new InterconnectLink());
    private pb::UnknownFieldSet _unknownFields;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pb::MessageParser<InterconnectLink> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::Tensorflow.DeviceAttributesReflection.Descriptor.MessageTypes[0]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public InterconnectLink() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public InterconnectLink(InterconnectLink other) : this() {
      deviceId_ = other.deviceId_;
      type_ = other.type_;
      strength_ = other.strength_;
      _unknownFields = pb::UnknownFieldSet.Clone(other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public InterconnectLink Clone() {
      return new InterconnectLink(this);
    }

    /// <summary>Field number for the "device_id" field.</summary>
    public const int DeviceIdFieldNumber = 1;
    private int deviceId_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int DeviceId {
      get { return deviceId_; }
      set {
        deviceId_ = value;
      }
    }

    /// <summary>Field number for the "type" field.</summary>
    public const int TypeFieldNumber = 2;
    private string type_ = "";
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string Type {
      get { return type_; }
      set {
        type_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "strength" field.</summary>
    public const int StrengthFieldNumber = 3;
    private int strength_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int Strength {
      get { return strength_; }
      set {
        strength_ = value;
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override bool Equals(object other) {
      return Equals(other as InterconnectLink);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool Equals(InterconnectLink other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if (DeviceId != other.DeviceId) return false;
      if (Type != other.Type) return false;
      if (Strength != other.Strength) return false;
      return Equals(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override int GetHashCode() {
      int hash = 1;
      if (DeviceId != 0) hash ^= DeviceId.GetHashCode();
      if (Type.Length != 0) hash ^= Type.GetHashCode();
      if (Strength != 0) hash ^= Strength.GetHashCode();
      if (_unknownFields != null) {
        hash ^= _unknownFields.GetHashCode();
      }
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void WriteTo(pb::CodedOutputStream output) {
      if (DeviceId != 0) {
        output.WriteRawTag(8);
        output.WriteInt32(DeviceId);
      }
      if (Type.Length != 0) {
        output.WriteRawTag(18);
        output.WriteString(Type);
      }
      if (Strength != 0) {
        output.WriteRawTag(24);
        output.WriteInt32(Strength);
      }
      if (_unknownFields != null) {
        _unknownFields.WriteTo(output);
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int CalculateSize() {
      int size = 0;
      if (DeviceId != 0) {
        size += 1 + pb::CodedOutputStream.ComputeInt32Size(DeviceId);
      }
      if (Type.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(Type);
      }
      if (Strength != 0) {
        size += 1 + pb::CodedOutputStream.ComputeInt32Size(Strength);
      }
      if (_unknownFields != null) {
        size += _unknownFields.CalculateSize();
      }
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(InterconnectLink other) {
      if (other == null) {
        return;
      }
      if (other.DeviceId != 0) {
        DeviceId = other.DeviceId;
      }
      if (other.Type.Length != 0) {
        Type = other.Type;
      }
      if (other.Strength != 0) {
        Strength = other.Strength;
      }
      _unknownFields = pb::UnknownFieldSet.MergeFrom(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(pb::CodedInputStream input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, input);
            break;
          case 8: {
            DeviceId = input.ReadInt32();
            break;
          }
          case 18: {
            Type = input.ReadString();
            break;
          }
          case 24: {
            Strength = input.ReadInt32();
            break;
          }
        }
      }
    }

  }

  public sealed partial class LocalLinks : pb::IMessage<LocalLinks> {
    private static readonly pb::MessageParser<LocalLinks> _parser = new pb::MessageParser<LocalLinks>(() => new LocalLinks());
    private pb::UnknownFieldSet _unknownFields;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pb::MessageParser<LocalLinks> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::Tensorflow.DeviceAttributesReflection.Descriptor.MessageTypes[1]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public LocalLinks() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public LocalLinks(LocalLinks other) : this() {
      link_ = other.link_.Clone();
      _unknownFields = pb::UnknownFieldSet.Clone(other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public LocalLinks Clone() {
      return new LocalLinks(this);
    }

    /// <summary>Field number for the "link" field.</summary>
    public const int LinkFieldNumber = 1;
    private static readonly pb::FieldCodec<global::Tensorflow.InterconnectLink> _repeated_link_codec
        = pb::FieldCodec.ForMessage(10, global::Tensorflow.InterconnectLink.Parser);
    private readonly pbc::RepeatedField<global::Tensorflow.InterconnectLink> link_ = new pbc::RepeatedField<global::Tensorflow.InterconnectLink>();
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public pbc::RepeatedField<global::Tensorflow.InterconnectLink> Link {
      get { return link_; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override bool Equals(object other) {
      return Equals(other as LocalLinks);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool Equals(LocalLinks other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if(!link_.Equals(other.link_)) return false;
      return Equals(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override int GetHashCode() {
      int hash = 1;
      hash ^= link_.GetHashCode();
      if (_unknownFields != null) {
        hash ^= _unknownFields.GetHashCode();
      }
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void WriteTo(pb::CodedOutputStream output) {
      link_.WriteTo(output, _repeated_link_codec);
      if (_unknownFields != null) {
        _unknownFields.WriteTo(output);
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int CalculateSize() {
      int size = 0;
      size += link_.CalculateSize(_repeated_link_codec);
      if (_unknownFields != null) {
        size += _unknownFields.CalculateSize();
      }
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(LocalLinks other) {
      if (other == null) {
        return;
      }
      link_.Add(other.link_);
      _unknownFields = pb::UnknownFieldSet.MergeFrom(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(pb::CodedInputStream input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, input);
            break;
          case 10: {
            link_.AddEntriesFrom(input, _repeated_link_codec);
            break;
          }
        }
      }
    }

  }

  public sealed partial class DeviceLocality : pb::IMessage<DeviceLocality> {
    private static readonly pb::MessageParser<DeviceLocality> _parser = new pb::MessageParser<DeviceLocality>(() => new DeviceLocality());
    private pb::UnknownFieldSet _unknownFields;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pb::MessageParser<DeviceLocality> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::Tensorflow.DeviceAttributesReflection.Descriptor.MessageTypes[2]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public DeviceLocality() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public DeviceLocality(DeviceLocality other) : this() {
      busId_ = other.busId_;
      numaNode_ = other.numaNode_;
      links_ = other.links_ != null ? other.links_.Clone() : null;
      _unknownFields = pb::UnknownFieldSet.Clone(other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public DeviceLocality Clone() {
      return new DeviceLocality(this);
    }

    /// <summary>Field number for the "bus_id" field.</summary>
    public const int BusIdFieldNumber = 1;
    private int busId_;
    /// <summary>
    /// Optional bus locality of device.  Default value of 0 means
    /// no specific locality.  Specific localities are indexed from 1.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int BusId {
      get { return busId_; }
      set {
        busId_ = value;
      }
    }

    /// <summary>Field number for the "numa_node" field.</summary>
    public const int NumaNodeFieldNumber = 2;
    private int numaNode_;
    /// <summary>
    /// Optional NUMA locality of device.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int NumaNode {
      get { return numaNode_; }
      set {
        numaNode_ = value;
      }
    }

    /// <summary>Field number for the "links" field.</summary>
    public const int LinksFieldNumber = 3;
    private global::Tensorflow.LocalLinks links_;
    /// <summary>
    /// Optional local interconnect links to other devices.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public global::Tensorflow.LocalLinks Links {
      get { return links_; }
      set {
        links_ = value;
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override bool Equals(object other) {
      return Equals(other as DeviceLocality);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool Equals(DeviceLocality other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if (BusId != other.BusId) return false;
      if (NumaNode != other.NumaNode) return false;
      if (!object.Equals(Links, other.Links)) return false;
      return Equals(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override int GetHashCode() {
      int hash = 1;
      if (BusId != 0) hash ^= BusId.GetHashCode();
      if (NumaNode != 0) hash ^= NumaNode.GetHashCode();
      if (links_ != null) hash ^= Links.GetHashCode();
      if (_unknownFields != null) {
        hash ^= _unknownFields.GetHashCode();
      }
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void WriteTo(pb::CodedOutputStream output) {
      if (BusId != 0) {
        output.WriteRawTag(8);
        output.WriteInt32(BusId);
      }
      if (NumaNode != 0) {
        output.WriteRawTag(16);
        output.WriteInt32(NumaNode);
      }
      if (links_ != null) {
        output.WriteRawTag(26);
        output.WriteMessage(Links);
      }
      if (_unknownFields != null) {
        _unknownFields.WriteTo(output);
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int CalculateSize() {
      int size = 0;
      if (BusId != 0) {
        size += 1 + pb::CodedOutputStream.ComputeInt32Size(BusId);
      }
      if (NumaNode != 0) {
        size += 1 + pb::CodedOutputStream.ComputeInt32Size(NumaNode);
      }
      if (links_ != null) {
        size += 1 + pb::CodedOutputStream.ComputeMessageSize(Links);
      }
      if (_unknownFields != null) {
        size += _unknownFields.CalculateSize();
      }
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(DeviceLocality other) {
      if (other == null) {
        return;
      }
      if (other.BusId != 0) {
        BusId = other.BusId;
      }
      if (other.NumaNode != 0) {
        NumaNode = other.NumaNode;
      }
      if (other.links_ != null) {
        if (links_ == null) {
          Links = new global::Tensorflow.LocalLinks();
        }
        Links.MergeFrom(other.Links);
      }
      _unknownFields = pb::UnknownFieldSet.MergeFrom(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(pb::CodedInputStream input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, input);
            break;
          case 8: {
            BusId = input.ReadInt32();
            break;
          }
          case 16: {
            NumaNode = input.ReadInt32();
            break;
          }
          case 26: {
            if (links_ == null) {
              Links = new global::Tensorflow.LocalLinks();
            }
            input.ReadMessage(Links);
            break;
          }
        }
      }
    }

  }

  public sealed partial class DeviceAttributes : pb::IMessage<DeviceAttributes> {
    private static readonly pb::MessageParser<DeviceAttributes> _parser = new pb::MessageParser<DeviceAttributes>(() => new DeviceAttributes());
    private pb::UnknownFieldSet _unknownFields;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pb::MessageParser<DeviceAttributes> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::Tensorflow.DeviceAttributesReflection.Descriptor.MessageTypes[3]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public DeviceAttributes() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public DeviceAttributes(DeviceAttributes other) : this() {
      name_ = other.name_;
      deviceType_ = other.deviceType_;
      memoryLimit_ = other.memoryLimit_;
      locality_ = other.locality_ != null ? other.locality_.Clone() : null;
      incarnation_ = other.incarnation_;
      physicalDeviceDesc_ = other.physicalDeviceDesc_;
      _unknownFields = pb::UnknownFieldSet.Clone(other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public DeviceAttributes Clone() {
      return new DeviceAttributes(this);
    }

    /// <summary>Field number for the "name" field.</summary>
    public const int NameFieldNumber = 1;
    private string name_ = "";
    /// <summary>
    /// Fully specified name of the device within a cluster.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string Name {
      get { return name_; }
      set {
        name_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "device_type" field.</summary>
    public const int DeviceTypeFieldNumber = 2;
    private string deviceType_ = "";
    /// <summary>
    /// String representation of device_type.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string DeviceType {
      get { return deviceType_; }
      set {
        deviceType_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "memory_limit" field.</summary>
    public const int MemoryLimitFieldNumber = 4;
    private long memoryLimit_;
    /// <summary>
    /// Memory capacity of device in bytes.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public long MemoryLimit {
      get { return memoryLimit_; }
      set {
        memoryLimit_ = value;
      }
    }

    /// <summary>Field number for the "locality" field.</summary>
    public const int LocalityFieldNumber = 5;
    private global::Tensorflow.DeviceLocality locality_;
    /// <summary>
    /// Platform-specific data about device that may be useful
    /// for supporting efficient data transfers.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public global::Tensorflow.DeviceLocality Locality {
      get { return locality_; }
      set {
        locality_ = value;
      }
    }

    /// <summary>Field number for the "incarnation" field.</summary>
    public const int IncarnationFieldNumber = 6;
    private ulong incarnation_;
    /// <summary>
    /// A device is assigned a global unique number each time it is
    /// initialized. "incarnation" should never be 0.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public ulong Incarnation {
      get { return incarnation_; }
      set {
        incarnation_ = value;
      }
    }

    /// <summary>Field number for the "physical_device_desc" field.</summary>
    public const int PhysicalDeviceDescFieldNumber = 7;
    private string physicalDeviceDesc_ = "";
    /// <summary>
    /// String representation of the physical device that this device maps to.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string PhysicalDeviceDesc {
      get { return physicalDeviceDesc_; }
      set {
        physicalDeviceDesc_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override bool Equals(object other) {
      return Equals(other as DeviceAttributes);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool Equals(DeviceAttributes other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if (Name != other.Name) return false;
      if (DeviceType != other.DeviceType) return false;
      if (MemoryLimit != other.MemoryLimit) return false;
      if (!object.Equals(Locality, other.Locality)) return false;
      if (Incarnation != other.Incarnation) return false;
      if (PhysicalDeviceDesc != other.PhysicalDeviceDesc) return false;
      return Equals(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override int GetHashCode() {
      int hash = 1;
      if (Name.Length != 0) hash ^= Name.GetHashCode();
      if (DeviceType.Length != 0) hash ^= DeviceType.GetHashCode();
      if (MemoryLimit != 0L) hash ^= MemoryLimit.GetHashCode();
      if (locality_ != null) hash ^= Locality.GetHashCode();
      if (Incarnation != 0UL) hash ^= Incarnation.GetHashCode();
      if (PhysicalDeviceDesc.Length != 0) hash ^= PhysicalDeviceDesc.GetHashCode();
      if (_unknownFields != null) {
        hash ^= _unknownFields.GetHashCode();
      }
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void WriteTo(pb::CodedOutputStream output) {
      if (Name.Length != 0) {
        output.WriteRawTag(10);
        output.WriteString(Name);
      }
      if (DeviceType.Length != 0) {
        output.WriteRawTag(18);
        output.WriteString(DeviceType);
      }
      if (MemoryLimit != 0L) {
        output.WriteRawTag(32);
        output.WriteInt64(MemoryLimit);
      }
      if (locality_ != null) {
        output.WriteRawTag(42);
        output.WriteMessage(Locality);
      }
      if (Incarnation != 0UL) {
        output.WriteRawTag(49);
        output.WriteFixed64(Incarnation);
      }
      if (PhysicalDeviceDesc.Length != 0) {
        output.WriteRawTag(58);
        output.WriteString(PhysicalDeviceDesc);
      }
      if (_unknownFields != null) {
        _unknownFields.WriteTo(output);
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int CalculateSize() {
      int size = 0;
      if (Name.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(Name);
      }
      if (DeviceType.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(DeviceType);
      }
      if (MemoryLimit != 0L) {
        size += 1 + pb::CodedOutputStream.ComputeInt64Size(MemoryLimit);
      }
      if (locality_ != null) {
        size += 1 + pb::CodedOutputStream.ComputeMessageSize(Locality);
      }
      if (Incarnation != 0UL) {
        size += 1 + 8;
      }
      if (PhysicalDeviceDesc.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(PhysicalDeviceDesc);
      }
      if (_unknownFields != null) {
        size += _unknownFields.CalculateSize();
      }
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(DeviceAttributes other) {
      if (other == null) {
        return;
      }
      if (other.Name.Length != 0) {
        Name = other.Name;
      }
      if (other.DeviceType.Length != 0) {
        DeviceType = other.DeviceType;
      }
      if (other.MemoryLimit != 0L) {
        MemoryLimit = other.MemoryLimit;
      }
      if (other.locality_ != null) {
        if (locality_ == null) {
          Locality = new global::Tensorflow.DeviceLocality();
        }
        Locality.MergeFrom(other.Locality);
      }
      if (other.Incarnation != 0UL) {
        Incarnation = other.Incarnation;
      }
      if (other.PhysicalDeviceDesc.Length != 0) {
        PhysicalDeviceDesc = other.PhysicalDeviceDesc;
      }
      _unknownFields = pb::UnknownFieldSet.MergeFrom(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(pb::CodedInputStream input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, input);
            break;
          case 10: {
            Name = input.ReadString();
            break;
          }
          case 18: {
            DeviceType = input.ReadString();
            break;
          }
          case 32: {
            MemoryLimit = input.ReadInt64();
            break;
          }
          case 42: {
            if (locality_ == null) {
              Locality = new global::Tensorflow.DeviceLocality();
            }
            input.ReadMessage(Locality);
            break;
          }
          case 49: {
            Incarnation = input.ReadFixed64();
            break;
          }
          case 58: {
            PhysicalDeviceDesc = input.ReadString();
            break;
          }
        }
      }
    }

  }

  #endregion

}

#endregion Designer generated code
